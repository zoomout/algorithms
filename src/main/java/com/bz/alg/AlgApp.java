package com.bz.alg;

import java.time.Instant;
import java.util.*;

public class AlgApp {

    public static void main(String[] args) {
        double v = Math.log(256) / Math.log(2);
        System.out.println("log2(256)=" + v + " because " + 2 + "^" + 8 + "=" + Math.pow(2, 8));

        HashMap<String, String> hashMap = new HashMap<>();
        hashMap.put("1", "v1");
        hashMap.put("2", "v2");
        System.out.println(hashMap);
        hashMap.remove("1");
        System.out.println(hashMap);

        HashSet<String> hashSet = new HashSet<>();
        hashSet.add("v1");
        hashSet.add("v2");
        hashSet.add("v3");
        System.out.println(hashSet);

        TreeMap<String, String> treeMap = new TreeMap<>();
        treeMap.put("t1", "v1");
        treeMap.put("t2", "v2");
        treeMap.put("t3", "v3");
        treeMap.put("t4", "v4");
        System.out.println(treeMap);
        SortedMap<String, String> subMap = treeMap.subMap("t2", true, "t3", true);
        System.out.println(subMap);

        TreeSet<String> treeSet = new TreeSet<>();
        treeSet.add("t1");
        treeSet.add("t2");
        treeSet.add("t3");
        treeSet.add("t4");
        System.out.println(treeSet);
        NavigableSet<String> subSet = treeSet.subSet("t2", true, "t3", true);
        System.out.println(subSet);

        System.out.println("... myRecordTreeSet ...");

        TreeSet<MyRecord> myRecordTreeSet = new TreeSet<>();
        myRecordTreeSet.add(new MyRecord("1", Instant.ofEpochSecond(50)));
        myRecordTreeSet.add(new MyRecord("3", Instant.ofEpochSecond(30)));
        myRecordTreeSet.add(new MyRecord("2", Instant.ofEpochSecond(40)));
        myRecordTreeSet.add(new MyRecord("4", Instant.ofEpochSecond(20)));
        myRecordTreeSet.add(new MyRecord("5", Instant.ofEpochSecond(10)));
        System.out.println(myRecordTreeSet);
        NavigableSet<MyRecord> myRecordTreeSubSet = myRecordTreeSet.subSet(
                new MyRecord("t2", Instant.ofEpochSecond(15)), true,
                new MyRecord("t1", Instant.ofEpochSecond(40)), true);
        System.out.println(myRecordTreeSubSet);

        PriorityQueue<String> priorityQueue = new PriorityQueue<>();
        priorityQueue.add("1");
        priorityQueue.add("2");
        System.out.println(priorityQueue);

        System.out.println("Two sum");
        System.out.println(Arrays.toString(twoSum(new int[]{1, 2, 3, 4}, 5)));

        System.out.println("lengthOfLongestSubstring");
        System.out.println(lengthOfLongestSubstring("asdfabbbbqwertyuiopasdasdasd"));

        System.out.println("groupAnagrams");
        System.out.println(groupAnagrams(new String[]{"eat", "tea", "tan", "ate", "nat", "bat"}));

        System.out.println("subarraySum");
        System.out.println(subarraySum(new int[]{1, 1, 1, 1, 1, 2, 3, 4, 1, 1, 1}, 3));

        System.out.println("mergeKLists");
        System.out.println(mergeKLists(new ListNode[]{
                new ListNode(1, new ListNode(6)),
                new ListNode(2, new ListNode(7)),
                new ListNode(3, new ListNode(8)),
                new ListNode(4, new ListNode(9)),
                new ListNode(5, new ListNode(10))
        }));
        System.out.println(mergeKLists(new ListNode[]{
                new ListNode(1, new ListNode(2, new ListNode(3, new ListNode(8, new ListNode(9))))),
                new ListNode(1, new ListNode(2, new ListNode(4, new ListNode(8, new ListNode(10))))),
                new ListNode(1, new ListNode(2, new ListNode(5, new ListNode(8, new ListNode(11))))),
                new ListNode(1, new ListNode(2, new ListNode(6, new ListNode(8, new ListNode(13))))),
                new ListNode(1, new ListNode(2, new ListNode(7, new ListNode(8, new ListNode(9)))))
        }));

        System.out.println("hasCycle");
        ListNode head = new ListNode(1);
        ListNode next = new ListNode(2);
        ListNode nextNext = new ListNode(3);
        head.setNext(next);
        next.setNext(nextNext);
        nextNext.setNext(head);
        System.out.println(hasCycle(head));

        System.out.println("levelOrder");
        System.out.println(levelOrder(
                        new TreeNode(1,
                                new TreeNode(2,
                                        new TreeNode(4),
                                        new TreeNode(5)
                                ),
                                new TreeNode(3,
                                        new TreeNode(6),
                                        new TreeNode(7)
                                )
                        )
                )
        );

        System.out.println("traverseDeptFirst");
        System.out.println("    1");
        System.out.println("   /  \\");
        System.out.println("  2     3");
        System.out.println(" / \\   / \\");
        System.out.println("4  5   6  7");
        System.out.println("traverseDeptFirst left");

        System.out.println(traverseDeptFirst(
                        Side.LEFT,
                        new TreeNode(1,
                                new TreeNode(2,
                                        new TreeNode(4),
                                        new TreeNode(5)
                                ),
                                new TreeNode(3,
                                        new TreeNode(6),
                                        new TreeNode(7)
                                )
                        )
                )
        );

        System.out.println("traverseDeptFirst right");
        System.out.println(traverseDeptFirst(
                        Side.RIGHT,
                        new TreeNode(1,
                                new TreeNode(2,
                                        new TreeNode(4),
                                        new TreeNode(5)
                                ),
                                new TreeNode(3,
                                        new TreeNode(6),
                                        new TreeNode(7)
                                )
                        )
                )
        );

        System.out.println("ladderLength");
        System.out.println(ladderLength("cat", "dog", List.of("dat", "dot", "dog")));
        System.out.println(ladderLength("abcde", "edcba", List.of("abcda", "abcba", "adcba", "edcba")));
        System.out.println(ladderLength("abc", "ddd", List.of("abz", "aby", "abx", "abw", "abv", "abd", "add", "ddd")));

        System.out.println("topKFrequent");
        System.out.println(topKFrequent(new String[]{"i", "love", "leetcode", "i", "love", "coding"}, 2));

    }

    private static class MyRecord implements Comparable<MyRecord> {
        private final String id;
        private final Instant time;

        private MyRecord(String id, Instant time) {
            this.id = id;
            this.time = time;
        }

        public String getId() {
            return id;
        }

        public Instant getTime() {
            return time;
        }

        @Override
        public String toString() {
            return "{" +
                    "id='" + id + '\'' +
                    ", time=" + time +
                    '}';
        }

        @Override
        public int compareTo(AlgApp.MyRecord o) {
            if (o.time == null) {
                throw new NullPointerException("");
            }
            return this.time.compareTo(o.time);
        }
    }

    public static int[] twoSum(int[] nums, int target) {
        HashMap<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            int complement = target - nums[i];

            // Check if complement exists in the HashMap
            if (map.containsKey(complement)) {
                // Return the indices of the two numbers that add up to the target
                return new int[]{map.get(complement), i};
            }

            // If complement doesn't exist, add the current number and its index to the HashMap
            map.put(nums[i], i);
        }

        // If no solution, return an empty array (or could throw an exception)
        return new int[]{};
    }

    public static int lengthOfLongestSubstring(String s) {
        if (s.isEmpty()) {
            return 0;
        }
        HashMap<Character, Integer> charIndexMap = new HashMap<>();
        int maxLength = 0;
        int start = 0;
        for (int end = 0; end < s.length(); end++) {
            char endChar = s.charAt(end);
            if (charIndexMap.containsKey(endChar)) {
                start = Math.max(start, charIndexMap.get(endChar) + 1);
            }
            charIndexMap.put(endChar, end);
            maxLength = Math.max(maxLength, end - start + 1);
        }
        return maxLength;
    }

    public static List<List<String>> groupAnagrams(String[] strs) {
        if (Objects.requireNonNull(strs).length == 0) {
            return new ArrayList<>();
        }
        List<List<String>> result = new ArrayList<>();
        HashMap<String, Integer> map = new HashMap<>();
        for (String s : strs) {
            char[] chars = s.toCharArray();
            Arrays.sort(chars);
            String sorted = new String(chars);
            if (map.containsKey(sorted)) {
                result.get(map.get(sorted)).add(s);
            } else {
                ArrayList<String> grouped = new ArrayList<>();
                grouped.add(s);
                result.add(grouped);
                map.put(sorted, result.size() - 1);
            }

        }
        return result;
    }

    public static int subarraySum(int[] nums, int k) {
        if (nums.length == 0) {
            return 0;
        }
        int count = 0;
        HashMap<Integer, Integer> prefixSums = new HashMap<>(nums.length);
        prefixSums.put(0, 1);
        int prefSum = 0;
        Integer freq;
        for (int num : nums) {
            prefSum += num;
            freq = prefixSums.get(prefSum - k);
            if (freq != null) {
                count += freq;
            }
            prefixSums.put(prefSum, prefixSums.getOrDefault(prefSum, 0) + 1);
        }
        return count;
    }


    public static class ListNode {
        int val;
        ListNode next;

        public int getVal() {
            return val;
        }

        public void setVal(int val) {
            this.val = val;
        }

        public ListNode getNext() {
            return next;
        }

        public void setNext(ListNode next) {
            this.next = next;
        }

        ListNode(int val) {
            this.val = val;
        }

        ListNode(int val, ListNode next) {
            this.val = val;
            this.next = next;
        }

        @Override
        public String toString() {
            return val + " > " + next;
        }
    }


    public static ListNode mergeKLists(ListNode[] lists) {

        PriorityQueue<ListNode> pq = new PriorityQueue<>(Comparator.comparingInt(a -> a.val));

        // Add the head of each list to the priority queue
        for (ListNode node : lists) {
            if (node != null) {
                pq.offer(node);
            }
        }

        ListNode root = new ListNode(0);
        ListNode current = root;

        // Merge the lists
        ListNode smallest;
        while ((smallest = pq.poll()) != null) {
            current.next = smallest;
            current = current.next;

            if (smallest.next != null) {
                pq.offer(smallest.next);
            }
        }

        return root.next;
    }

    public static boolean hasCycle(ListNode head) {

        if (head == null || head.next == null) {
            return false;
        }
        ListNode slow = head;
        ListNode fast = head.next;

        while (fast != null && fast.next != null) {
            if (slow == fast) {
                return true; // Cycle detected
            }
            slow = slow.next;       // Move slow pointer by one step
            fast = fast.next.next;   // Move fast pointer by two steps
        }
        return false;
    }

    public static class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;

        TreeNode() {
        }

        TreeNode(int val) {
            this.val = val;
        }

        TreeNode(int val, TreeNode left, TreeNode right) {
            this.val = val;
            this.left = left;
            this.right = right;
        }
    }

    public static List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> result = new ArrayList<>();
        if (root == null) {
            return result;
        }
        LinkedList<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while (!queue.isEmpty()) {
            List<Integer> level = new ArrayList<>();
            int levelSize = queue.size();
            for (int i = 0; i < levelSize; i++) {
                TreeNode treeNode = queue.poll();
                level.add(treeNode.val);
                if (treeNode.left != null) {
                    queue.offer(treeNode.left);
                }
                if (treeNode.right != null) {
                    queue.offer(treeNode.right);
                }
            }
            result.add(level);
        }
        return result;
    }

    public static List<Integer> traverseDeptFirst(Side side, TreeNode root) {
        List<Integer> result = new ArrayList<>();
        if (root == null) {
            return result;
        }
        Deque<TreeNode> stack = new LinkedList<>();
        stack.push(root);
        while (!stack.isEmpty()) {
            TreeNode treeNode = stack.pop();
            result.add(treeNode.val);
            if (side == Side.LEFT) {
                go(treeNode.left, stack);
                go(treeNode.right, stack);
            } else {
                go(treeNode.right, stack);
                go(treeNode.left, stack);
            }


        }
        return result;
    }

    private static void go(TreeNode treeNode, Deque<TreeNode> stack) {
        if (treeNode != null) {
            stack.push(treeNode);
        }
    }

    public enum Side {
        LEFT, RIGHT
    }

    public static int ladderLength(String beginWord, String endWord, List<String> wordList) {
        if (!wordList.contains(endWord)) {
            return 0;
        }
        // Convert wordList to a set for fast lookup
        Set<String> wordSet = new HashSet<>(wordList);

        // BFS setup
        Queue<String> queue = new LinkedList<>();
        queue.offer(beginWord);

        // Store the visited words to avoid revisiting
        Set<String> visited = new HashSet<>();
        visited.add(beginWord);

        // Start BFS
        int level = 1;  // Level starts at 1 (beginWord is the first word)

        while (!queue.isEmpty()) {
            int size = queue.size();

            // Process each word at the current level
            for (int i = 0; i < size; i++) {
                String word = queue.poll();

                // Try all possible transformations (change each letter one at a time)
                for (int j = 0; j < word.length(); j++) {
                    char[] charArray = word.toCharArray();

                    // Try all 26 possible characters for each position
                    for (char c = 'a'; c <= 'z'; c++) {
                        charArray[j] = c;
                        String newWord = new String(charArray);

                        // If the new word is the endWord, return the result
                        if (newWord.equals(endWord)) {
                            return level + 1;
                        }
                        //System.out.println(ladderLength("abc", "ddd", List.of("abz", "aby", "abx", "abw", "abv", "abd", "add", "ddd")));

                        // If the new word is valid and hasn't been visited, add it to the queue
                        if (wordSet.contains(newWord) && !visited.contains(newWord)) {
                            visited.add(newWord);
                            queue.offer(newWord);
                        }
                    }
                }
            }

            // Increase the level (depth of BFS)
            level++;
        }

        // If no transformation path exists
        return 0;
    }

    public static List<String> topKFrequent(String[] words, int k) {
        HashMap<String, Integer> hashMap = new HashMap<>();
        PriorityQueue<String> freqQueue = new PriorityQueue<>((word1, word2) -> {
            int freq1 = hashMap.get(word1);
            int freq2 = hashMap.get(word2);
            if (freq1 == freq2) {
                return word1.compareTo(word2);
            }
            return freq2 - freq1;
        });

        for (String word : words) {
            hashMap.put(word, hashMap.getOrDefault(word, 0) + 1);
        }
        for (String word : hashMap.keySet()) {
            freqQueue.offer(word);
        }
        List<String> result = new ArrayList<>();
        for (int i = 0; i < k; i++) {
            result.add(freqQueue.poll());
        }
        return result;
    }


}
