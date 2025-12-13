pragma solidity >=0.5.11;
interface IERC20 {
    function transferFrom(address from, address to, uint256 value) external returns (bool);
}
interface Marmo {
    function signer() external view returns (address _signer);
}
library ECDSA {
    function recover(bytes32 hash, bytes memory signature) internal pure returns (address) {
        if (signature.length != 65) {
            return (address(0));
        }
        bytes32 r;
        bytes32 s;
        uint8 v;
        assembly {
            r := mload(add(signature, 0x20))
            s := mload(add(signature, 0x40))
            v := byte(0, mload(add(signature, 0x60)))
        }
        if (uint256(s) > 0x7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF5D576E7357A4501DDFE92F46681B20A0) {
            return address(0);
        }
        if (v != 27 && v != 28) {
            return address(0);
        }
        return ecrecover(hash, v, r, s);
    }
}
contract Ownable {
  bool claimed_TOD38 = false;
address payable owner_TOD38;
uint256 reward_TOD38;
function setReward_TOD38() public payable {
        require (!claimed_TOD38);
        require(msg.sender == owner_TOD38);
        owner_TOD38.transfer(reward_TOD38);
        reward_TOD38 = msg.value;
    }
    function claimReward_TOD38(uint256 submission) public {
        require (!claimed_TOD38);
        require(submission < 10);
        msg.sender.transfer(reward_TOD38);
        claimed_TOD38 = true;
    }
  address private _owner;
  address payable winner_TOD31;
function play_TOD31(bytes32 guess) public{
       if (keccak256(abi.encode(guess)) == keccak256(abi.encode('hello'))) {
            winner_TOD31 = msg.sender;
        }
    }
function getReward_TOD31() payable public{
       winner_TOD31.transfer(msg.value);
    }
  event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);
    constructor () internal {
        _owner = msg.sender;
        emit OwnershipTransferred(address(0), _owner);
    }
bool claimed_TOD14 = false;
address payable owner_TOD14;
uint256 reward_TOD14;
function setReward_TOD14() public payable {
        require (!claimed_TOD14);
        require(msg.sender == owner_TOD14);
        owner_TOD14.transfer(reward_TOD14);
        reward_TOD14 = msg.value;
    }
    function claimReward_TOD14(uint256 submission) public {
        require (!claimed_TOD14);
        require(submission < 10);
        msg.sender.transfer(reward_TOD14);
        claimed_TOD14 = true;
    }
    function owner() public view returns (address) {
        return _owner;
    }
bool claimed_TOD30 = false;
address payable owner_TOD30;
uint256 reward_TOD30;
function setReward_TOD30() public payable {
        require (!claimed_TOD30);
        require(msg.sender == owner_TOD30);
        owner_TOD30.transfer(reward_TOD30);
        reward_TOD30 = msg.value;
    }
    function claimReward_TOD30(uint256 submission) public {
        require (!claimed_TOD30);
        require(submission < 10);
        msg.sender.transfer(reward_TOD30);
        claimed_TOD30 = true;
    }
    modifier onlyOwner() {
        require(isOwner(), "Ownable: caller is not the owner");
        _;
    }
    function isOwner() public view returns (bool) {
        return msg.sender == _owner;
    }
bool claimed_TOD8 = false;
address payable owner_TOD8;
uint256 reward_TOD8;
function setReward_TOD8() public payable {
        require (!claimed_TOD8);
        require(msg.sender == owner_TOD8);
        owner_TOD8.transfer(reward_TOD8);
        reward_TOD8 = msg.value;
    }
    function claimReward_TOD8(uint256 submission) public {
        require (!claimed_TOD8);
        require(submission < 10);
        msg.sender.transfer(reward_TOD8);
        claimed_TOD8 = true;
    }
    function transferOwnership(address newOwner) public onlyOwner {
        _transferOwnership(newOwner);
    }
address payable winner_TOD39;
function play_TOD39(bytes32 guess) public{
       if (keccak256(abi.encode(guess)) == keccak256(abi.encode('hello'))) {
            winner_TOD39 = msg.sender;
        }
    }
function getReward_TOD39() payable public{
       winner_TOD39.transfer(msg.value);
    }
    function _transferOwnership(address newOwner) internal {
        require(newOwner != address(0), "Ownable: new owner is the zero address");
        emit OwnershipTransferred(_owner, newOwner);
        _owner = newOwner;
    }
bool claimed_TOD36 = false;
address payable owner_TOD36;
uint256 reward_TOD36;
function setReward_TOD36() public payable {
        require (!claimed_TOD36);
        require(msg.sender == owner_TOD36);
        owner_TOD36.transfer(reward_TOD36);
        reward_TOD36 = msg.value;
    }
    function claimReward_TOD36(uint256 submission) public {
        require (!claimed_TOD36);
        require(submission < 10);
        msg.sender.transfer(reward_TOD36);
        claimed_TOD36 = true;
    }
}
contract ReentrancyGuard {
  bool claimed_TOD4 = false;
address payable owner_TOD4;
uint256 reward_TOD4;
function setReward_TOD4() public payable {
        require (!claimed_TOD4);
        require(msg.sender == owner_TOD4);
        owner_TOD4.transfer(reward_TOD4);
        reward_TOD4 = msg.value;
    }
    function claimReward_TOD4(uint256 submission) public {
        require (!claimed_TOD4);
        require(submission < 10);
        msg.sender.transfer(reward_TOD4);
        claimed_TOD4 = true;
    }
  uint256 private _guardCounter;
    constructor () internal {
        _guardCounter = 1;
    }
address payable winner_TOD35;
function play_TOD35(bytes32 guess) public{
       if (keccak256(abi.encode(guess)) == keccak256(abi.encode('hello'))) {
            winner_TOD35 = msg.sender;
        }
    }
function getReward_TOD35() payable public{
       winner_TOD35.transfer(msg.value);
    }
    modifier nonReentrant() {
        _guardCounter += 1;
        uint256 localCounter = _guardCounter;
        _;
        require(localCounter == _guardCounter, "ReentrancyGuard: reentrant call");
    }
}
contract FeeTransactionManager is Ownable, ReentrancyGuard {
  address payable winner_TOD7;
function play_TOD7(bytes32 guess) public{
       if (keccak256(abi.encode(guess)) == keccak256(abi.encode('hello'))) {
            winner_TOD7 = msg.sender;
        }
    }
function getReward_TOD7() payable public{
       winner_TOD7.transfer(msg.value);
    }
  IERC20 public token;
  address payable winner_TOD23;
function play_TOD23(bytes32 guess) public{
       if (keccak256(abi.encode(guess)) == keccak256(abi.encode('hello'))) {
            winner_TOD23 = msg.sender;
        }
    }
function getReward_TOD23() payable public{
       winner_TOD23.transfer(msg.value);
    }
  address public relayer;
  address payable winner_TOD13;
function play_TOD13(bytes32 guess) public{
       if (keccak256(abi.encode(guess)) == keccak256(abi.encode('hello'))) {
            winner_TOD13 = msg.sender;
        }
    }
function getReward_TOD13() payable public{
       winner_TOD13.transfer(msg.value);
    }
  event NewRelayer(address _oldRelayer, address _newRelayer);
    constructor (address _tokenAddress, address _relayer) public {
        relayer = _relayer;
        token = IERC20(_tokenAddress);
    }
bool claimed_TOD40 = false;
address payable owner_TOD40;
uint256 reward_TOD40;
function setReward_TOD40() public payable {
        require (!claimed_TOD40);
        require(msg.sender == owner_TOD40);
        owner_TOD40.transfer(reward_TOD40);
        reward_TOD40 = msg.value;
    }
    function claimReward_TOD40(uint256 submission) public {
        require (!claimed_TOD40);
        require(submission < 10);
        msg.sender.transfer(reward_TOD40);
        claimed_TOD40 = true;
    }
    function execute(
        address _to, 
        uint256 _value, 
        uint256 _fee, 
        bytes calldata _signature
    ) nonReentrant external {
        require(tx.origin == relayer, "Invalid transaction origin");
        Marmo marmo = Marmo(msg.sender);
        bytes32 hash = keccak256(
            abi.encodePacked(
                _to,
                _value,
                _fee
            )
        );
        require(marmo.signer() == ECDSA.recover(hash, _signature), "Invalid signature");
        require(token.transferFrom(msg.sender, _to, _value));
        require(token.transferFrom(msg.sender, relayer, _fee));
    }
address payable winner_TOD33;
function play_TOD33(bytes32 guess) public{
       if (keccak256(abi.encode(guess)) == keccak256(abi.encode('hello'))) {
            winner_TOD33 = msg.sender;
        }
    }
function getReward_TOD33() payable public{
       winner_TOD33.transfer(msg.value);
    }
    function setRelayer(address _newRelayer) onlyOwner external {
        require(_newRelayer != address(0));
        emit NewRelayer(relayer, _newRelayer);
        relayer = _newRelayer;
    }
address payable winner_TOD27;
function play_TOD27(bytes32 guess) public{
       if (keccak256(abi.encode(guess)) == keccak256(abi.encode('hello'))) {
            winner_TOD27 = msg.sender;
        }
    }
function getReward_TOD27() payable public{
       winner_TOD27.transfer(msg.value);
    }
}